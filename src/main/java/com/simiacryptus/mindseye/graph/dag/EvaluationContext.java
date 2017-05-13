package com.simiacryptus.mindseye.graph.dag;

import com.simiacryptus.mindseye.net.NNResult;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class EvaluationContext {

  public final Map<UUID, NNResult> cache = new HashMap<>();

}
