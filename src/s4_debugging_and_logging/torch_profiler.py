import torch
import torchvision.models as models
from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

model = models.resnet18()
DEVICE = torch.device("cuda" if True else "cpu")

model = model.to(DEVICE)
inputs = torch.randn(5, 3, 224, 224).to(DEVICE)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

#prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)
#prof.start()
#model(inputs)
#prof.stop()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
#prof.export_chrome_trace("trace.json")

