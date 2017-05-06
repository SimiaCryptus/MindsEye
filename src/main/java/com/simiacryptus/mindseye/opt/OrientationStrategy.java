package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;

/**
 * Created by Andrew Charneski on 5/6/2017.
 */
public interface OrientationStrategy {

    DeltaSet orient(Trainable.PointSample measurement, TrainingMonitor monitor);
}
