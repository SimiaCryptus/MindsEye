package com.simiacryptus.mindseye.net;

import com.simiacryptus.mindseye.core.delta.NNResult;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Created by Andrew Charneski on 4/25/2017.
 */
public class EvaluationContext {

  public final Map<UUID, NNResult> cache = new HashMap<>();

}
