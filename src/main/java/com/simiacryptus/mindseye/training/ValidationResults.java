package com.simiacryptus.mindseye.training;

import java.util.List;

import com.simiacryptus.util.ml.Tensor;

public class ValidationResults {
  public final List<Tensor> outputs;
  public final double rms;

  public ValidationResults(final List<Tensor> outputs, final double rms) {
    super();
    this.outputs = outputs;
    this.rms = rms;
  }
}
