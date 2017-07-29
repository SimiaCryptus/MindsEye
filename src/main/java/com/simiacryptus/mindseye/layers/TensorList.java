package com.simiacryptus.mindseye.layers;

import com.simiacryptus.util.ml.Tensor;

import java.util.stream.Stream;

public interface TensorList {
    Tensor get(int i);

    int length();

    Stream<Tensor> stream();
}
