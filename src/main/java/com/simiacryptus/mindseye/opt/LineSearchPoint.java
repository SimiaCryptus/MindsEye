package com.simiacryptus.mindseye.opt;

/**
 * Created by Andrew Charneski on 5/9/2017.
 */
public class LineSearchPoint {
    public final Trainable.PointSample point;
    public final double derivative;

    public LineSearchPoint(Trainable.PointSample point, double derivative) {
        this.point = point;
        this.derivative = derivative;
    }
}
