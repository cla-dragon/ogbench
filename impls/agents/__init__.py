from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.hiql_nnx import HIQLAgent as HIQLAgentNNX
from agents.hiql_nnx import create_hiql_agent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    hiql_nnx=HIQLAgentNNX,
)

create_funcs = dict(
    hiql_nnx=create_hiql_agent,
)