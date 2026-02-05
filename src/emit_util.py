from typing import Callable,Optional

StatusCB = Callable[[dict],None]

def emit(status_cb:Optional[StatusCB],event:dict):
    if status_cb:
        status_cb(event)
    else:
        print(event.get("message",""))
