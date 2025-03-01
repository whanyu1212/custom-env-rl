import numpy as np
import matplotlib.pyplot as plt
import sys

# sys.path.insert(0, "..")
from src.custom_trading import CustomTradingEnv, Actions


# Generate synthetic price data (random walk)
def generate_price_data(days=100, volatility=0.50, initial_price=100):
    """Generate synthetic price data using random walk."""
    price = initial_price
    prices = [price]

    for _ in range(days - 1):
        change_percent = np.random.normal(0, volatility)
        price = price * (1 + change_percent)
        prices.append(price)

    return np.array(prices)


# Create environment
def test_random_actions():
    # Generate synthetic price data
    prices = generate_price_data(days=500, volatility=0.05, initial_price=100)

    # Initialize environment
    env = CustomTradingEnv(
        data=prices,
        window_size=20,
        initial_balance=10000,
        transaction_cost_pct=0.01,
        slippage=0.005,
    )

    # Initialize tracking variables
    portfolio_values = []
    actions_taken = []
    positions = []

    # Reset environment
    observation, _ = env.reset()

    # Run simulation with random actions
    done = False
    while not done:
        # Generate random action
        action = {
            "action_type": np.random.randint(0, 7),  # Random action type
            "amount": np.array([np.random.random()]),  # Random percentage 0-1
            "price_modifier": np.array(
                [0.7 + 0.6 * np.random.random()]
            ),  # Random modifier 0.7-1.3
        }

        # Take action
        observation, reward, done, _, info = env.step(action)

        # Track results
        portfolio_values.append(info["portfolio_value"])
        actions_taken.append(Actions(action["action_type"]).name)
        positions.append(info["shares_held"])

        # Print progress periodically
        # if env.current_step % 10 == 0:
        print(
            f"Step: {env.current_step}, Portfolio: ${info['portfolio_value']:.2f}, "
            f"Action: {Actions(action['action_type']).name}, Shares: {info['shares_held']:.2f}"
        )

    # Print final results
    print("\nSimulation Complete!")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    print(
        f"Return: {(portfolio_values[-1] - env.initial_balance) / env.initial_balance:.2%}"
    )

    # Plot results
    plot_results(prices, portfolio_values, actions_taken, positions)


def plot_results(prices, portfolio_values, actions_taken, positions):
    """Plot the results of the trading simulation."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot price data
    ax1.plot(prices)
    ax1.set_title("Price Data")
    ax1.set_ylabel("Price")

    # Plot portfolio value
    ax2.plot(portfolio_values)
    ax2.set_title("Portfolio Value")
    ax2.set_ylabel("Value ($)")

    # Plot position size
    ax3.plot(positions)
    ax3.set_title("Position Size (Shares)")
    ax3.set_ylabel("Shares")
    ax3.set_xlabel("Time Step")

    # Add markers for buy/sell actions
    for i, action in enumerate(actions_taken):
        if action == "Buy" or action == "LimitBuy":
            ax1.scatter(i, prices[i], color="green", marker="^", alpha=0.7)
        elif action == "Sell" or action == "LimitSell":
            ax1.scatter(i, prices[i], color="red", marker="v", alpha=0.7)
        elif action == "StopLoss":
            ax1.scatter(i, prices[i], color="purple", marker="x", alpha=0.7)
        elif action == "TakeProfit":
            ax1.scatter(i, prices[i], color="blue", marker="o", alpha=0.7)

    plt.tight_layout()
    plt.savefig("trading_simulation.png")
    plt.show()


if __name__ == "__main__":
    test_random_actions()
