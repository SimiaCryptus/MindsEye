package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;

/**
 * Created by Andrew Charneski on 5/6/2017.
 */
public interface ScalingStrategy {

    Trainable.PointSample step(Trainable subject, DeltaSet direction, Trainable.PointSample measurement, TrainingMonitor monitor);
}
