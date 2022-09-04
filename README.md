# SimpleFDL
Simple Federated Learning Framework.

## Workflow
Cient:
    send join info -> receive train info -> train -> send model -> calculate $q$ -> send $q$ -> receive $kl$ -> select -> 
    send $sum(q)$ -> next round

Server:
    waiting client -> send train info -> receive model -> aggregate -> receive $q$ -> calculate $kl$ -> send $kl$ -> receive $sum(q)$ -> update $p$ -> send model -> next round

## Client -> Server
data format: { typ: str, client_id: int, content: }
typ:
    'join': join the fdl.
    'model': send model weights.
    'q': send q.
    'sumq': send $sum(q)$.

## Server -> Client
data format: { typ: str, content: }
typ:
    'start': send train info, tell client to start.
    'kl': send 'kl'.

