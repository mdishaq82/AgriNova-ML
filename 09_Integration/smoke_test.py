from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'06_Final_Model'))
from prediction_pipeline import predict_crop
sample={'N':90,'P':42,'K':43,'temperature':20.879744,'humidity':82.002744,'ph':6.502985,'rainfall':202.935536}
print(predict_crop(sample))
