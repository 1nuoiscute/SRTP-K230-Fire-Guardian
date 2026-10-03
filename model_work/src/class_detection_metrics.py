"""Score only classes with actual frame truth, never infer absent-class negatives."""
from data_integrity import matches


def scores_by_known_class(truth,predictions):
    result={}
    for cls,boxes in truth.items():
        score=matches(boxes,[p['xyxy'] for p in predictions if p['class']==cls])
        denominator=2*score['tp']+score['fp']+score['fn']
        result[cls]={**score,'f1':2*score['tp']/denominator if denominator else None}
    return result
