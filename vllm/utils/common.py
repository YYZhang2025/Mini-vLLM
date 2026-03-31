import os
import urllib.request

import rich
import torch
from filelock import FileLock
from rich.console import Console
from rich.table import Table


def print_master(s: str = "", type: str = "info", **kwargs):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if rank == 0:
        if type == "info":
            rich.print(f"[blue][info] {s}[/blue]", **kwargs)
        elif type == "error":
            rich.print(f"[red][error] {s}[/red]", **kwargs)
        elif type == "success":
            rich.print(f"[green][success] {s}[/green]", **kwargs)
        else:
            rich.print(s, **kwargs)
