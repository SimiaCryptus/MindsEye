package com.simiacryptus.mindseye.training;

import java.util.List;

import com.simiacryptus.mindseye.math.NDArray;

public class ValidationResults {
  public final List<NDArray> outputs;
  public final double rms;
  public ValidationResults(List<NDArray> outputs, double rms) {
    super();
    this.outputs = outputs;
    this.rms = rms;
  }
}