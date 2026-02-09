
import argparse
from typing import Any, Callable, Dict, List, Literal, Mapping, TypeAlias


Config = Mapping[str, Any]

ARG_RESOLVER: TypeAlias = Callable[[Config], Any]
ARG_VALIDATOR: TypeAlias = Callable[[Config], None]

ARG_RESOLVERS_COLL: TypeAlias = Dict[str, ARG_VALIDATOR]
ARG_VALIDATORS_COLL: TypeAlias = List[ARG_VALIDATOR]

SubsetArg = Literal['train', 'test', 'val', 'all']