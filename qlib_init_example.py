import os
import qlib
from qlib.constant import REG_CN

# 设置Qlib数据目录路径
# 请将下面的路径替换为你的实际数据路径
provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")

# 初始化Qlib
qlib.init(provider_uri=provider_uri, region=REG_CN)

print(f"Qlib initialized with data from: {provider_uri}")
print("You can now use Qlib for data query, model training, and backtesting!")
