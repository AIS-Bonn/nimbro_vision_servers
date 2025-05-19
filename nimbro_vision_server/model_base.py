from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseModel(ABC):
    @classmethod
    @abstractmethod
    def get_available_flavors(cls) -> List[str]:
        """
        Return a list of valid flavor names (e.g. ["tiny","base","large"])
        that can be passed to load().
        Class method because an instance is not needed/available yet.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Return the name of the model family.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict:
        """
        Return a dictionary describing the status of the model.
        """
        pass

    @abstractmethod
    def load(self, payload: Dict[str, Any]) -> None:
        """
        Instantiate/configure the model according to the payload.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Tear down the model (free GPU memory, etc.)."""
        pass

    @abstractmethod
    def preprocess(self, payload: Dict[str, Any]) -> Any:
        """
        Convert the raw request payload (e.g. JSON dict with image bytes,
        parameters, etc.) into whatever inputs the model needs.
        """
        pass

    @abstractmethod
    def infer(self, inputs: Any) -> Any:
        """Run the forward pass and return raw model outputs."""
        pass

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """
        Convert raw outputs into the API’s response format
        (e.g. JSON-serializable dict of boxes, masks, classes).
        """
        pass