"""Metrics for operational cascade cohorts."""



def operational_auc(video_scores):
    """Compute AUC for exactly the videos processed by one stage.

    ``None`` represents N/A when the stage has no videos or only one class.
    """
    from sklearn.metrics import roc_auc_score
    labels = [video.label for video in video_scores]
    if any(label is None for label in labels) or len(set(labels)) < 2:
        return None
    scores = [video.score for video in video_scores]
    return float(roc_auc_score(labels, scores))


def final_classification_metrics(final_videos):
    """Compute video-level ACC, FNR/FPR and legacy FAR/FRR aliases.

    Fake is the positive class. FAR is fake classified as REAL, while FRR is
    real classified as FAKE. A missing class produces ``None`` for its rate.
    """
    final_videos = tuple(final_videos)
    if not final_videos:
        raise ValueError('Final metrics require at least one video.')

    correct = 0
    false_accepts = 0
    false_rejects = 0
    fake_count = 0
    real_count = 0

    for video in final_videos:
        if video.decision not in ('REAL', 'FAKE'):
            raise ValueError('Final metrics cannot contain UNCERTAIN decisions.')
        predicted_label = 1 if video.decision == 'FAKE' else 0
        if video.label not in (0, 1):
            raise ValueError('Final video labels must be binary.')

        correct += int(predicted_label == video.label)
        if video.label == 1:
            fake_count += 1
            false_accepts += int(predicted_label == 0)
        else:
            real_count += 1
            false_rejects += int(predicted_label == 1)

    fnr = None if fake_count == 0 else false_accepts / fake_count
    fpr = None if real_count == 0 else false_rejects / real_count
    return {
        'acc': correct / len(final_videos),
        'fnr': fnr,
        'fpr': fpr,
        'far': fnr,
        'frr': fpr,
        'video_count': len(final_videos),
        'real_count': real_count,
        'fake_count': fake_count,
    }
