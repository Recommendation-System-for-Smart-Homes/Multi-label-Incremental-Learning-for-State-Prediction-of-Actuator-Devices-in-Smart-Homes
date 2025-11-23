from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class CodeDTO:
    code: str
    value: str
    dpId: int = 0
    service: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "dpId": self.dpId,
            "value": self.value,
            "service": self.service
        }


@dataclass
class MessageDTO:
    status: List[CodeDTO]

    def __eq__(self, other) -> bool:
        return isinstance(other, MessageDTO) and self.status == other.status

    def to_dict(self) -> dict:
        return {
            "status": [i.to_dict() for i in self.status],
        }


@dataclass
class DeviceStatusDTO:
    environment: str
    devId: str
    device: str
    space: str
    message: Optional[MessageDTO] = None
    alert: Optional[MessageDTO] = None
    productKey: str = ""
    sensorType: str = ""
    timeStamp: str = ""
    hardware: Optional[str] = None
    timestamp: Optional[str] = None
    hardware_name: Optional[str] = None
    nodeID: Optional[str] = None
    time_metrics: Dict = field(default_factory=dict)
    service: str = ""

    def to_dict(self) -> dict:
        return {
            "environment": self.environment,
            "devId": self.devId,
            "device": self.device,
            "productKey": self.productKey,
            "space": self.space,
            "message": self.message.to_dict() if self.message else None,
            "alert": self.alert.to_dict() if self.alert else None,
            "sensorType": self.sensorType,
            "timeStamp": self.timeStamp,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "hardware_name": self.hardware_name,
            "nodeID": self.nodeID,
            "time_metrics": self.time_metrics,
            "service": self.service,
        }