package com.simiacryptus.mindseye.opt;

/**
 * Created by Andrew Charneski on 5/6/2017.
 */
public interface OrientationStrategy {

    LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor);
}
