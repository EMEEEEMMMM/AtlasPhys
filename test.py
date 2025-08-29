import queue, threading

q = queue.Queue(maxsize=3)  # 容量为3的队列

def producer():
    for i in range(5):
        q.put(f"Task-{i}")
        print(f"Produced: Task-{i}")

def consumer():
    while True:
        item = q.get()
        print(f"Consumed: {item}")
        q.task_done()

threading.Thread(target=producer, daemon=True).start()
threading.Thread(target=consumer, daemon=True).start()
q.join()  # 等待所有任务完成