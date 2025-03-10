from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class SystemConfig(Base):
    """System configuration parameters"""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    value = Column(Text, nullable=True)
    value_type = Column(String(20), nullable=False)  # string, float, int, json, etc.
    description = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def get_typed_value(self):
        """Convert stored string value to its proper type"""
        if self.value_type == "float":
            return float(self.value)
        elif self.value_type == "int":
            return int(self.value)
        elif self.value_type == "bool":
            return self.value.lower() == "true"
        elif self.value_type == "json":
            import json
            return json.loads(self.value)
        else:
            return self.value

class BatteryConfig(Base):
    """Battery configuration"""
    __tablename__ = "battery_config"
    
    id = Column(Integer, primary_key=True)
    capacity = Column(Float, nullable=False)  # kWh
    max_power = Column(Float, nullable=False)  # kW
    efficiency_charge = Column(Float, nullable=False)  # decimal
    efficiency_discharge = Column(Float, nullable=False)  # decimal
    min_soc = Column(Float, nullable=False)  # decimal
    max_soc = Column(Float, nullable=False)  # decimal
    degradation_cost = Column(Float, nullable=True)  # $/kWh
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class PVConfig(Base):
    """PV system configuration"""
    __tablename__ = "pv_config"
    
    id = Column(Integer, primary_key=True)
    capacity = Column(Float, nullable=False)  # kWp
    inverter_capacity = Column(Float, nullable=True)  # kW
    inverter_efficiency = Column(Float, nullable=True)  # decimal
    orientation = Column(String(50), nullable=True)  # e.g., "South"
    tilt = Column(Float, nullable=True)  # degrees
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class GridConfig(Base):
    """Grid connection configuration"""
    __tablename__ = "grid_config"
    
    id = Column(Integer, primary_key=True)
    max_import = Column(Float, nullable=False)  # kW
    max_export = Column(Float, nullable=True)  # kW
    import_tariff_type = Column(String(50), nullable=True)  # e.g., "flat", "tou", "dynamic"
    export_tariff_type = Column(String(50), nullable=True)  # e.g., "flat", "tou", "dynamic"
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class TariffSchedule(Base):
    """Time-of-use tariff schedule"""
    __tablename__ = "tariff_schedule"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    tariff_type = Column(String(50), nullable=False)  # "import" or "export"
    day_type = Column(String(50), nullable=False)  # "weekday", "weekend", "all"
    start_hour = Column(Integer, nullable=False)  # 0-23
    end_hour = Column(Integer, nullable=False)  # 0-23
    rate = Column(Float, nullable=False)  # $/kWh
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class DeferrableLoad(Base):
    """Deferrable load configuration"""
    __tablename__ = "deferrable_load"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    device_id = Column(String(100), nullable=False)  # Home Assistant entity_id or ESPHome device_id
    device_type = Column(String(50), nullable=False)  # "homeassistant" or "esphome"
    power = Column(Float, nullable=False)  # kW
    min_runtime = Column(Integer, nullable=True)  # minutes
    max_runtime = Column(Integer, nullable=True)  # minutes
    required_runtime = Column(Integer, nullable=True)  # minutes per day
    earliest_start = Column(Integer, nullable=True)  # hour of day (0-23)
    latest_end = Column(Integer, nullable=True)  # hour of day (0-23)
    priority = Column(Integer, nullable=True)  # 1-10, higher is more important
    enabled = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class EnergyData(Base):
    """Historical energy data"""
    __tablename__ = "energy_data"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    load_power = Column(Float, nullable=True)  # kW
    pv_power = Column(Float, nullable=True)  # kW
    battery_power = Column(Float, nullable=True)  # kW (positive = charging, negative = discharging)
    battery_soc = Column(Float, nullable=True)  # percentage
    grid_power = Column(Float, nullable=True)  # kW (positive = import, negative = export)
    grid_import_price = Column(Float, nullable=True)  # $/kWh
    grid_export_price = Column(Float, nullable=True)  # $/kWh
    
    def __repr__(self):
        return f"<EnergyData(timestamp='{self.timestamp}', load_power={self.load_power}, pv_power={self.pv_power})>"

class ForecastData(Base):
    """Forecast data"""
    __tablename__ = "forecast_data"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    forecast_type = Column(String(50), nullable=False)  # "load", "pv", "price"
    forecast_method = Column(String(50), nullable=False)  # "arima", "random_forest", "pattern", etc.
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    time_step = Column(Integer, nullable=False)  # minutes
    data = Column(JSON, nullable=False)  # JSON array of forecast values
    accuracy = Column(Float, nullable=True)  # RMSE or other accuracy metric
    
    def __repr__(self):
        return f"<ForecastData(type='{self.forecast_type}', method='{self.forecast_method}', start='{self.start_time}')>"

class OptimizationResult(Base):
    """Optimization results"""
    __tablename__ = "optimization_result"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    time_step = Column(Integer, nullable=False)  # minutes
    objective = Column(String(50), nullable=False)  # "cost", "self_consumption", etc.
    status = Column(String(50), nullable=False)  # "optimal", "infeasible", etc.
    cost = Column(Float, nullable=True)  # $
    self_consumption = Column(Float, nullable=True)  # percentage
    peak_grid_power = Column(Float, nullable=True)  # kW
    battery_cycles = Column(Float, nullable=True)  # cycles
    grid_power = Column(JSON, nullable=True)  # JSON array of grid power values
    battery_power = Column(JSON, nullable=True)  # JSON array of battery power values
    battery_soc = Column(JSON, nullable=True)  # JSON array of battery SOC values
    device_schedules = Column(JSON, nullable=True)  # JSON object of device schedules
    
    def __repr__(self):
        return f"<OptimizationResult(created='{self.created_at}', objective='{self.objective}', status='{self.status}')>"

class Recommendation(Base):
    """Energy usage recommendations"""
    __tablename__ = "recommendation"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    device_id = Column(String(100), nullable=True)  # Optional reference to a specific device
    recommendation_type = Column(String(50), nullable=False)  # "load_shifting", "battery_usage", etc.
    priority = Column(Integer, nullable=False)  # 1-10, higher is more important
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    potential_savings = Column(Float, nullable=True)  # $ or kWh
    start_time = Column(DateTime, nullable=True)  # When to start the recommended action
    end_time = Column(DateTime, nullable=True)  # When to end the recommended action
    implemented = Column(Boolean, default=False)  # Whether the recommendation was implemented
    
    def __repr__(self):
        return f"<Recommendation(type='{self.recommendation_type}', title='{self.title}')>"

class UserFeedback(Base):
    """User feedback on recommendations"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True)
    recommendation_id = Column(Integer, ForeignKey("recommendation.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    helpful = Column(Boolean, nullable=False)
    comments = Column(Text, nullable=True)
    
    recommendation = relationship("Recommendation")
    
    def __repr__(self):
        return f"<UserFeedback(recommendation_id={self.recommendation_id}, helpful={self.helpful})>"

