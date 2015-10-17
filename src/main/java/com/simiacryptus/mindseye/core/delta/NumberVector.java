package com.simiacryptus.mindseye.core.delta;

import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.TrainingContext;

public class NumberVector {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TrainingContext.class);

  private final double[] array;

  public <T extends Number> NumberVector(final double[] values) {
    this.array = Arrays.copyOf(values, values.length);
  }

  public double[] getArray() {
    return this.array;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("DeltaFlushBuffer [");
    builder.append(Arrays.toString(getArray()));
    builder.append("]");
    return builder.toString();
  }

}
