package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaSet;

public interface Trainable {
    class PointSample {
        public final DeltaSet delta;
        public final DeltaSet weights;
        public final double value;

        public PointSample(DeltaSet delta, DeltaSet weights, double value) {
            this.delta = delta;
            this.weights = weights;
            this.value = value;
        }
    }
    PointSample measure();
    void resetToFull();
    void resetSampling();
}
