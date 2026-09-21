from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'06_Final_Model'))
from prediction_pipeline import predict_crop

def test_prediction():
    sample={'N':90,'P':42,'K':43,'temperature':20.879744,'humidity':82.002744,'ph':6.502985,'rainfall':202.935536}
    result=predict_crop(sample)
    assert result['crop']
    assert 0 <= result['accuracy'] <= 1
