package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;

public class GradientDescent implements OrientationStrategy {

    @Override
    public LineSearchCursor orient(Trainable subject, Trainable.PointSample measurement, TrainingMonitor monitor) {
        DeltaSet direction = measurement.delta.scale(-1);
        return new SimpleLineSearchCursor(measurement, direction, subject);
    }

}
