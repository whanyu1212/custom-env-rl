import numpy as np
import gymnasium as gym
from enum import IntEnum
from gymnasium import spaces
from typing import Tuple, Union, Optional, Dict


class Actions(IntEnum):
    Hold = 0
    Buy = 1
    Sell = 2
    LimitBuy = 3  # Limit buy order, to buy at a specific price
    LimitSell = 4  # Limit sell order, to sell at a specific price
    StopLoss = 5  # Stop loss order, to sell if price drops below a certain level
    TakeProfit = 6  # Take profit order, to sell if price rises above a certain level


class CustomTradingEnv(gym.Env):
    def __init__(
        self,
        data: np.array,
        window_size: int,
        initial_balance: Union[int, float],
        transaction_cost_pct: float,
        slippage: float,
    ):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage = slippage
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.order_expiration_steps = 5  # Limit orders auto expire after 5 steps

        # Initialize order tracking
        self.pending_orders = (
            []
        )  # this needs to be handled before we take any steps, e.g. close pending orders after expiry

        self.stop_loss_orders = []
        self.take_profit_orders = []
        self.executed_orders = []  # Track executed orders for history

        # Define action space
        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(7),  # All action types
                "amount": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),  # Percentage
                "price_modifier": spaces.Box(
                    low=0.7, high=1.3, shape=(1,), dtype=np.float32
                ),  # Price level modifier for limit orders
            }
        )

        # Define observation space
        obs_dim = (
            self.window_size + 5
        )  # 5 portfolio metrics, refer to the portfolio_info array
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )  # refer to the _get_observation method

    def _get_portfolio_value(self) -> float:
        """Get the total portfolio value at the current step.

        Returns:
            float: portfolio value
        """
        if self.current_step < len(self.data):
            current_price = self.data[self.current_step]

            # Note: shares_held already excludes reserved shares, so we use just shares_held
            portfolio_value = self.balance + (self.shares_held * current_price)

            # Add value of shares in risk management orders
            for order in self.stop_loss_orders:
                portfolio_value += order["shares"] * current_price

            for order in self.take_profit_orders:
                portfolio_value += order["shares"] * current_price

            # Include value of pending limit sell orders (but not at their limit prices)
            # Using current price is more realistic for actual portfolio value
            for order in self.pending_orders:
                if order["type"] == "limit_sell":
                    portfolio_value += order["shares"] * current_price

            return portfolio_value
        return self.balance

    def _get_reserved_shares(self) -> int:
        """Get the total number of shares reserved in pending orders that
        are not yet executed

        Returns:
            int: total number of reserved shares in integer
        """
        reserved_in_stop_loss = sum(order["shares"] for order in self.stop_loss_orders)
        reserved_in_take_profit = sum(
            order["shares"] for order in self.take_profit_orders
        )
        reserved_in_limit_sell = sum(
            order["shares"]
            for order in self.pending_orders
            if order["type"] == "limit_sell"
        )
        return reserved_in_stop_loss + reserved_in_take_profit + reserved_in_limit_sell

    def _get_observation(self) -> np.array:
        """The observation space for the agent. we combine 2 sources
        of information: price history and portfolio information.

        Returns:
            np.array: array concatenating price history and portfolio info
        """

        # Last window_size prices
        start_idx = max(0, self.current_step - self.window_size + 1)
        prices = self.data[start_idx : self.current_step + 1]

        # Pad with zeros if we don't have enough history yet
        if len(prices) < self.window_size:
            padding = np.zeros(self.window_size - len(prices))
            prices = np.concatenate((padding, prices))

        # Normalize prices
        if len(prices) > 0 and prices[0] != 0:
            normalized_prices = prices / prices[0]
        else:
            normalized_prices = prices

        # Total position = currently held + reserved
        total_position = self.shares_held + self._get_reserved_shares()

        # Portfolio information - avoid division by zero with safer ratios
        if total_position > 0:
            stop_loss_ratio = (
                sum(order["shares"] for order in self.stop_loss_orders) / total_position
            )
            take_profit_ratio = (
                sum(order["shares"] for order in self.take_profit_orders)
                / total_position
            )
        else:
            stop_loss_ratio = 0.0
            take_profit_ratio = 0.0

        portfolio_info = np.array(
            [
                self.balance / self.initial_balance,  # Normalized cash
                total_position
                * self.data[self.current_step]
                / self.initial_balance,  # Position value
                1.0 if total_position > 0 else 0.0,  # Position flag
                stop_loss_ratio,  # Ratio of shares covered by stop loss
                take_profit_ratio,  # Ratio of shares covered by take profit
            ]
        )

        # Combine price history with portfolio info
        return np.concatenate((normalized_prices, portfolio_info))

    def _process_pending_orders(self, current_price: float) -> None:
        """Process pending limit orders to fulfill and check for expiration.

        Args:
            current_price (float): current price at the current step
        """
        remaining_orders = []

        for order in self.pending_orders:
            # initialize boolean flag
            executed = False

            # Check if order should be executed
            if order["type"] == "limit_buy" and current_price <= order["price"]:
                # Apply slight slippage - buy at the better of limit price or current price
                execution_price = min(order["price"], current_price)
                # Limit buy executed
                self.shares_held += order["shares"]
                executed = True

                # Record execution
                self.executed_orders.append(
                    {
                        "step": self.current_step,
                        "type": "limit_buy_executed",
                        "shares": order["shares"],
                        "price": execution_price,
                        "cost": order["shares"]
                        * order["price"]
                        * (1 + self.transaction_cost_pct),
                    }
                )

            elif order["type"] == "limit_sell" and current_price >= order["price"]:
                # Apply slight slippage - sell at the better of limit price or current price
                execution_price = max(order["price"], current_price)
                # Limit sell executed
                revenue = (
                    order["shares"] * execution_price * (1 - self.transaction_cost_pct)
                )
                self.balance += revenue
                executed = True

                # Record execution
                self.executed_orders.append(
                    {
                        "step": self.current_step,
                        "type": "limit_sell_executed",
                        "shares": order["shares"],
                        "price": execution_price,
                        "revenue": revenue,
                    }
                )

            elif self.current_step - order["placed_at"] > self.order_expiration_steps:
                # Order expired, return resources
                if order["type"] == "limit_buy":
                    # Return reserved funds
                    self.balance += (
                        order["shares"]
                        * order["price"]
                        * (1 + self.transaction_cost_pct)
                    )
                elif order["type"] == "limit_sell":
                    # Return reserved shares
                    self.shares_held += order["shares"]

                # Record expiration
                self.executed_orders.append(
                    {
                        "step": self.current_step,
                        "type": f"{order['type']}_expired",
                        "shares": order["shares"],
                        "price": order["price"],
                    }
                )

                executed = True

            # If not executed, keep in pending orders
            if not executed:
                remaining_orders.append(order)

        # Update pending orders list
        self.pending_orders = remaining_orders

    def _process_risk_management_orders(self, current_price: float) -> None:
        """Process stop loss and take profit orders based on current price.

        Args:
            current_price (float): current price at the current step
        """
        # Process stop loss orders
        remaining_stop_loss = []
        for order in self.stop_loss_orders:
            if current_price <= order["price"]:
                # Stop loss triggered
                adjusted_price = current_price * (1 - self.slippage)  # Apply slippage
                revenue = (
                    order["shares"] * adjusted_price * (1 - self.transaction_cost_pct)
                )
                self.balance += revenue
                # Note: We already reduced shares_held when placing the order
            else:
                # Stop loss not triggered
                remaining_stop_loss.append(order)
        self.stop_loss_orders = remaining_stop_loss

        # Process take profit orders
        remaining_take_profit = []
        for order in self.take_profit_orders:
            if current_price >= order["price"]:
                # Take profit triggered
                adjusted_price = current_price * (1 - self.slippage)  # Apply slippage
                revenue = (
                    order["shares"] * adjusted_price * (1 - self.transaction_cost_pct)
                )
                self.balance += revenue
                # Note: We already reduced shares_held when placing the order
            else:
                # Take profit not triggered
                remaining_take_profit.append(order)
        self.take_profit_orders = remaining_take_profit

    def _process_market_order(
        self, action_type: int, current_price: float, amount_pct: float
    ) -> None:
        """Process market orders to buy or sell shares.

        Args:
            action_type (int): 1 for buy, 2 for sell
            current_price (float): current price at the current step
            amount_pct (float): percentage of shares to buy or sell
        """
        # Apply slippage to current price
        if action_type == Actions.Buy:
            # Price moves up slightly due to slippage when buying
            adjusted_price = current_price * (1 + self.slippage)

            # Calculate shares to buy based on available balance
            max_shares = self.balance / (
                adjusted_price * (1 + self.transaction_cost_pct)
            )
            shares_to_buy = int(max_shares * amount_pct)  # Convert to integer shares

            # Skip if zero shares (prevent tiny orders)
            if shares_to_buy <= 0:
                return

            # Calculate cost with integer shares
            cost = shares_to_buy * adjusted_price * (1 + self.transaction_cost_pct)

            # Execute order if we have enough balance
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
            # Record execution
            self.executed_orders.append(
                {
                    "step": self.current_step,
                    "type": "market_buy_executed",
                    "shares": shares_to_buy,
                    "price": adjusted_price,
                    "cost": cost,
                }
            )

        elif action_type == Actions.Sell:
            # Price moves down slightly due to slippage when selling
            adjusted_price = current_price * (1 - self.slippage)

            # Calculate shares to sell (as integer)
            shares_to_sell = int(self.shares_held * amount_pct)

            # Skip if zero shares
            if shares_to_sell <= 0:
                return

            # Calculate revenue
            revenue = shares_to_sell * adjusted_price * (1 - self.transaction_cost_pct)

            # Execute order if we have shares to sell
            if shares_to_sell > 0:
                self.shares_held -= shares_to_sell
                self.balance += revenue

            # Record execution
            self.executed_orders.append(
                {
                    "step": self.current_step,
                    "type": "market_sell_executed",
                    "shares": shares_to_sell,
                    "price": adjusted_price,
                    "revenue": revenue,
                }
            )

    def _process_limit_order(
        self,
        action_type: int,
        current_price: float,
        amount_pct: float,
        price_modifier: float,
    ) -> None:
        """Place limit orders to buy or sell shares at a specific price.

        Args:
            action_type (int): 3 for limit buy, 4 for limit sell
            current_price (float): current price at the current step
            amount_pct (float): percentage of shares to buy or sell
            price_modifier (float): price level modifier for limit orders
        """
        if action_type == Actions.LimitBuy:
            # Limit buy order (typically price_modifier < 1.0)
            limit_price = current_price * price_modifier

            # Calculate shares and cost (integer shares)
            max_shares = self.balance / (limit_price * (1 + self.transaction_cost_pct))
            shares_to_buy = int(max_shares * amount_pct)

            # Skip if zero shares
            if shares_to_buy <= 0:
                return

            cost = shares_to_buy * limit_price * (1 + self.transaction_cost_pct)

            # Add to pending orders if we have funds
            if cost <= self.balance:
                self.pending_orders.append(
                    {
                        "type": "limit_buy",
                        "shares": shares_to_buy,  # Integer shares
                        "price": limit_price,
                        "placed_at": self.current_step,
                    }
                )
                # Reserve the funds
                self.balance -= cost

        elif action_type == Actions.LimitSell and self.shares_held > 0:
            # Limit sell order (typically price_modifier > 1.0)
            limit_price = current_price * price_modifier

            # Calculate shares to sell (integer)
            shares_to_sell = int(self.shares_held * amount_pct)

            # Skip if zero shares
            if shares_to_sell <= 0:
                return

            # Add to pending orders if we have shares
            if shares_to_sell > 0:
                self.pending_orders.append(
                    {
                        "type": "limit_sell",
                        "shares": shares_to_sell,  # Integer shares
                        "price": limit_price,
                        "placed_at": self.current_step,
                    }
                )
                # Reserve the shares
                self.shares_held -= shares_to_sell

    def step(self, action: dict) -> Tuple[np.array, float, bool, dict]:
        """Take a step in the environment based on the action and
        observation space.

        Args:
            action (dict): action to take during the step

        Returns:
            Tuple[np.array, float, bool, dict]: _description_
        """
        current_price = self.data[self.current_step]
        action_type = action["action_type"]
        amount_pct = float(action["amount"][0])
        price_modifier = float(action["price_modifier"][0])

        # Store current portfolio value for reward calculation
        prev_portfolio_value = self._get_portfolio_value()

        # Process existing orders first
        self._process_pending_orders(current_price)
        self._process_risk_management_orders(current_price)

        # Calculate available (unreserved) shares for risk management orders
        reserved_shares = self._get_reserved_shares()
        available_shares = max(0, self.shares_held - reserved_shares)

        # If action == 0, do nothing
        if action_type == Actions.Hold:
            pass

        # If action == 1 or 2, process market order
        elif action_type == Actions.Buy or action_type == Actions.Sell:
            self._process_market_order(action_type, current_price, amount_pct)

        # If action == 3 or 4, process limit order
        elif action_type == Actions.LimitBuy or action_type == Actions.LimitSell:
            self._process_limit_order(
                action_type, current_price, amount_pct, price_modifier
            )

        elif action_type == Actions.StopLoss and available_shares > 0:
            # Stop loss is placed below current price (for long positions)
            stop_price = current_price * min(
                0.99, price_modifier
            )  # Ensure it's below current price
            shares_covered = int(
                available_shares * amount_pct
            )  # Integer shares from available shares

            if shares_covered > 0:  # Ensure we're placing an order with >0 shares
                self.stop_loss_orders.append(
                    {
                        "shares": shares_covered,
                        "price": stop_price,
                        "placed_at": self.current_step,
                    }
                )
                # Reserve these shares - they can't be used by other orders
                self.shares_held -= shares_covered

        elif action_type == Actions.TakeProfit and available_shares > 0:
            # Take profit is placed above current price (for long positions)
            take_profit_price = current_price * max(
                1.01, price_modifier
            )  # Ensure it's above current price
            shares_covered = int(
                available_shares * amount_pct
            )  # Integer shares from available shares

            if shares_covered > 0:  # Ensure we're placing an order with >0 shares
                self.take_profit_orders.append(
                    {
                        "shares": shares_covered,
                        "price": take_profit_price,
                        "placed_at": self.current_step,
                    }
                )
                # Reserve these shares - they can't be used by other orders
                self.shares_held -= shares_covered

        # After processing actions and orders, move to next time step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Calculate reward based on portfolio value change
        new_portfolio_value = self._get_portfolio_value()

        # Avoid division by zero and excessive rewards
        if prev_portfolio_value > 0:
            reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            # Cap extreme rewards
            reward = max(min(reward, 1.0), -1.0)
        else:
            reward = 0

        # Add penalty for trading too frequently
        if action_type != Actions.Hold:
            reward -= self.transaction_cost_pct * 0.1  # Small penalty for trading

        # Get observation
        obs = self._get_observation()

        # Create info dictionary
        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "reserved_shares": self._get_reserved_shares(),
            "current_price": current_price,
            "returns": reward,
            "pending_orders": (
                len(self.pending_orders) if hasattr(self, "pending_orders") else 0
            ),
            "stop_loss_orders": len(self.stop_loss_orders),
            "take_profit_orders": len(self.take_profit_orders),
        }

        return (
            obs,
            reward,
            done,
            False,
            info,
        )  # truncate is not applicable here, just using False as a placeholder

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.array, dict]:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int], optional): Defaults to None.
            options (Optional[Dict], optional): Defaults to None.

        Returns:
            Tuple[np.array, dict]: observation and empty info dictionary
        """
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = (
            self.window_size
        )  # Start after enough history for observation

        # Define observation space (previously missing)
        # Calculate observation dimensionality: price history + portfolio info
        obs_dim = self.window_size + 5  # 5 portfolio metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Reset order tracking
        if hasattr(self, "pending_orders"):
            self.pending_orders = []
        self.stop_loss_orders = []
        self.take_profit_orders = []

        return self._get_observation(), {}  # Return initial observation and empty info
