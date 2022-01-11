
import time

def test_loop(total=10):
    from tqdm.auto import trange
    for _ in trange(total):
        time.sleep(1)
        
test_loop(10)
