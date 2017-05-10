package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.net.DeltaBuffer;
import com.simiacryptus.mindseye.net.DeltaSet;

import java.util.List;
import java.util.stream.IntStream;

public class SimpleLineSearchCursor implements LineSearchCursor {
    public final Trainable.PointSample origin;
    public final DeltaSet direction;
    public final Trainable subject;

    public SimpleLineSearchCursor(Trainable.PointSample origin, DeltaSet direction, Trainable subject) {
        this.origin = origin;
        this.direction = direction;
        this.subject = subject;
    }

    @Override
    public LineSearchPoint step(double alpha, TrainingMonitor monitor) {
        origin.weights.vector().stream().forEach(d -> d.overwrite());
        direction.vector().stream().forEach(d -> d.write(alpha));
        Trainable.PointSample sample = subject.measure();
        return new LineSearchPoint(sample, dot(direction.vector(), sample.delta.vector()));
    }

    protected static double dot(List<DeltaBuffer> a, List<DeltaBuffer> b) {
        assert (a.size() == b.size());
        return IntStream.range(0, a.size()).mapToDouble(i -> a.get(i).dot(b.get(i))).sum();
    }

}
