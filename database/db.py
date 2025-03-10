from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
import os
from database.models import Base

# Get database URL from environment or use default SQLite database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///energy_management.db")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def init_db():
    """Initialize the database"""
    Base.metadata.create_all(engine)
    
    # Create default configurations if they don't exist
    from database.models import SystemConfig, BatteryConfig, PVConfig, GridConfig, TariffSchedule
    
    with get_db_session() as session:
        # Check if we need to initialize default data
        if session.query(SystemConfig).count() == 0:
            # System config
            session.add(SystemConfig(name="time_horizon", value="24", value_type="int", description="Optimization time horizon in hours"))
            session.add(SystemConfig(name="time_step", value="15", value_type="int", description="Time step in minutes"))
            session.add(SystemConfig(name="optimization_objective", value="cost", value_type="string", description="Default optimization objective"))
            
            # Battery config
            session.add(BatteryConfig(
                capacity=10.0,
                max_power=5.0,
                efficiency_charge=0.95,
                efficiency_discharge=0.95,
                min_soc=0.2,
                max_soc=0.95,
                degradation_cost=0.05
            ))
            
            # PV config
            session.add(PVConfig(
                capacity=5.0,
                inverter_capacity=5.0,
                inverter_efficiency=0.97,
                orientation="South",
                tilt=30.0
            ))
            
            # Grid config
            session.add(GridConfig(
                max_import=10.0,
                max_export=5.0,
                import_tariff_type="tou",
                export_tariff_type="flat"
            ))
            
            # Default TOU tariffs
            # Weekday tariffs
            session.add(TariffSchedule(name="Off-peak", tariff_type="import", day_type="weekday", start_hour=0, end_hour=7, rate=0.08))
            session.add(TariffSchedule(name="Mid-peak", tariff_type="import", day_type="weekday", start_hour=7, end_hour=16, rate=0.12))
            session.add(TariffSchedule(name="On-peak", tariff_type="import", day_type="weekday", start_hour=16, end_hour=21, rate=0.24))
            session.add(TariffSchedule(name="Off-peak", tariff_type="import", day_type="weekday", start_hour=21, end_hour=24, rate=0.08))
            
            # Weekend tariffs
            session.add(TariffSchedule(name="Off-peak", tariff_type="import", day_type="weekend", start_hour=0, end_hour=24, rate=0.08))
            
            # Export tariffs
            session.add(TariffSchedule(name="Export", tariff_type="export", day_type="all", start_hour=0, end_hour=24, rate=0.05))
            
            print("Initialized database with default configurations")

def get_system_config():
    """Get all system configurations as a dictionary"""
    from database.models import SystemConfig
    
    with get_db_session() as session:
        configs = session.query(SystemConfig).all()
        return {config.name: config.get_typed_value() for config in configs}

def get_battery_config():
    """Get battery configuration"""
    from database.models import BatteryConfig
    
    with get_db_session() as session:
        config = session.query(BatteryConfig).first()
        if config:
            return {
                "capacity": config.capacity,
                "max_power": config.max_power,
                "efficiency_charge": config.efficiency_charge,
                "efficiency_discharge": config.efficiency_discharge,
                "min_soc": config.min_soc,
                "max_soc": config.max_soc,
                "degradation_cost": config.degradation_cost
            }
        return None

def get_pv_config():
    """Get PV configuration"""
    from database.models import PVConfig
    
    with get_db_session() as session:
        config = session.query(PVConfig).first()
        if config:
            return {
                "capacity": config.capacity,
                "inverter_capacity": config.inverter_capacity,
                "inverter_efficiency": config.inverter_efficiency,
                "orientation": config.orientation,
                "tilt": config.tilt
            }
        return None

def get_grid_config():
    """Get grid configuration"""
    from database.models import GridConfig
    
    with get_db_session() as session:
        config = session.query(GridConfig).first()
        if config:
            return {
                "max_import": config.max_import,
                "max_export": config.max_export,
                "import_tariff_type": config.import_tariff_type,
                "export_tariff_type": config.export_tariff_type
            }
        return None

def get_tariff_schedules():
    """Get tariff schedules"""
    from database.models import TariffSchedule
    
    with get_db_session() as session:
        tariffs = session.query(TariffSchedule).all()
        result = {
            "import": {
                "weekday": [],
                "weekend": [],
                "all": []
            },
            "export": {
                "weekday": [],
                "weekend": [],
                "all": []
            }
        }
        
        for tariff in tariffs:
            result[tariff.tariff_type][tariff.day_type].append({
                "name": tariff.name,
                "start_hour": tariff.start_hour,
                "end_hour": tariff.end_hour,
                "rate": tariff.rate
            })
        
        return result

def get_deferrable_loads():
    """Get all deferrable loads"""
    from database.models import DeferrableLoad
    
    with get_db_session() as session:
        loads = session.query(DeferrableLoad).filter_by(enabled=True).all()
        return [{
            "id": load.id,
            "name": load.name,
            "device_id": load.device_id,
            "device_type": load.device_type,
            "power": load.power,
            "min_runtime": load.min_runtime,
            "max_runtime": load.max_runtime,
            "required_runtime": load.required_runtime,
            "earliest_start": load.earliest_start,
            "latest_end": load.latest_end,
            "priority": load.priority
        } for load in loads]

def save_energy_data(data):
    """Save energy data to database"""
    from database.models import EnergyData
    
    with get_db_session() as session:
        energy_data = EnergyData(**data)
        session.add(energy_data)

def save_forecast(forecast_type, forecast_method, start_time, end_time, time_step, data, accuracy=None):
    """Save forecast data to database"""
    from database.models import ForecastData
    
    with get_db_session() as session:
        forecast = ForecastData(
            forecast_type=forecast_type,
            forecast_method=forecast_method,
            start_time=start_time,
            end_time=end_time,
            time_step=time_step,
            data=data,
            accuracy=accuracy
        )
        session.add(forecast)

def save_optimization_result(result):
    """Save optimization result to database"""
    from database.models import OptimizationResult
    
    with get_db_session() as session:
        optimization = OptimizationResult(**result)
        session.add(optimization)

def save_recommendation(recommendation):
    """Save recommendation to database"""
    from database.models import Recommendation
    
    with get_db_session() as session:
        rec = Recommendation(**recommendation)
        session.add(rec)
        session.flush()
        return rec.id

def get_recent_recommendations(limit=10):
    """Get recent recommendations"""
    from database.models import Recommendation
    from sqlalchemy import desc
    
    with get_db_session() as session:
        recommendations = session.query(Recommendation).order_by(desc(Recommendation.created_at)).limit(limit).all()
        return [{
            "id": rec.id,
            "created_at": rec.created_at,
            "device_id": rec.device_id,
            "recommendation_type": rec.recommendation_type,
            "priority": rec.priority,
            "title": rec.title,
            "description": rec.description,
            "potential_savings": rec.potential_savings,
            "start_time": rec.start_time,
            "end_time": rec.end_time,
            "implemented": rec.implemented
        } for rec in recommendations]

def get_historical_energy_data(start_time, end_time):
    """Get historical energy data for a time period"""
    from database.models import EnergyData
    
    with get_db_session() as session:
        data = session.query(EnergyData).filter(
            EnergyData.timestamp >= start_time,
            EnergyData.timestamp <= end_time
        ).order_by(EnergyData.timestamp).all()
        
        return [{
            "timestamp": record.timestamp,
            "load_power": record.load_power,
            "pv_power": record.pv_power,
            "battery_power": record.battery_power,
            "battery_soc": record.battery_soc,
            "grid_power": record.grid_power,
            "grid_import_price": record.grid_import_price,
            "grid_export_price": record.grid_export_price
        } for record in data]

def get_latest_forecast(forecast_type):
    """Get the latest forecast of a specific type"""
    from database.models import ForecastData
    from sqlalchemy import desc
    
    with get_db_session() as session:
        forecast = session.query(ForecastData).filter_by(
            forecast_type=forecast_type
        ).order_by(desc(ForecastData.created_at)).first()
        
        if forecast:
            return {
                "id": forecast.id,
                "created_at": forecast.created_at,
                "forecast_type": forecast.forecast_type,
                "forecast_method": forecast.forecast_method,
                "start_time": forecast.start_time,
                "end_time": forecast.end_time,
                "time_step": forecast.time_step,
                "data": forecast.data,
                "accuracy": forecast.accuracy
            }
        return None

