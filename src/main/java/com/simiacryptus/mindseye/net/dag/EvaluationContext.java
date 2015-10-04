package com.simiacryptus.mindseye.net.dag;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import com.simiacryptus.mindseye.deltas.NNResult;

public class EvaluationContext {

  public final Map<UUID, NNResult> cache = new HashMap<>();

}
