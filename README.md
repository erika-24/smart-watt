# SmartWatt - Low Wattage Energy Management System

SmartWatt is an intelligent energy management system designed specifically for low wattage devices such as LEDs, PWM motors, fans, and sensors. It optimizes energy usage, reduces costs, and maximizes self-consumption of renewable energy.

## Features

- **Real-time monitoring** of energy production and consumption for low wattage devices
- **Smart scheduling** of controllable devices like LED strips, fans, and motors
- **Multiple optimization algorithms** including Linear Programming, Stochastic Gradient Descent, and Genetic Algorithms
- **Energy forecasting** based on historical data and weather predictions
- **Device control** for various low wattage devices
- **Visualization** of energy flows and consumption patterns

## Supported Devices

- LED lighting (RGB strips, individual LEDs)
- Fans and cooling systems
- PWM-controlled motors and pumps
- Servo motors
- Sensors and controllers
- Other low wattage devices

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see `.env.example`)
4. Run the application: `python app.py`

## Usage

1. Access the dashboard at `http://localhost:5000`
2. Configure your devices and energy sources
3. Run optimization to generate efficient schedules
4. Apply schedules to your devices

## Technologies

- Python with Flask for the backend
- Pandas and NumPy for data processing
- CVXPY for linear programming optimization
- Custom SGD and Genetic Algorithm implementations
- Plotly for data visualization
- Bootstrap for the frontend

## License

MIT License

