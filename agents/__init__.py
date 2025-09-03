from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.ppo import PPOAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    ppo=PPOAgent,
)
