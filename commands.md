#
## Singularity set up like in [README](Readme.md)
## Test API KEY
```
python -c "
import logging; logging.basicConfig(level=logging.INFO)
from agents.llm import AittaClient, AittaConfig
from pydantic import BaseModel
class Ping(BaseModel):
    answer: str
c = AittaClient(AittaConfig())
print(c.structured(system='Reply tersely.', user='Say hello.', schema=Ping))
"
```
## Test 
```
pip install -e '.[chem,llm]'
python -c "import chemprop, rdkit, lightgbm, openai; print('ok')"
 ```
## Test
```

```
