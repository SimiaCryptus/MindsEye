# PipelineNetwork
## Debug
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.network.PipelineNetwork",
      "id": "a864e734-2f23-44db-97c1-504000002aff",
      "isFrozen": false,
      "name": "PipelineNetwork/a864e734-2f23-44db-97c1-504000002aff",
      "inputs": [
        "f58be311-6abc-4576-b9d6-640ea38e9d53",
        "b5a5d600-2727-49d5-a2d1-fa7084cd66c0"
      ],
      "nodes": {
        "848b0b20-a4eb-4aaf-b695-d5a17b2abc82": "a864e734-2f23-44db-97c1-504000002b00",
        "23bbba8a-a406-467d-8ee6-925057fc17a0": "a864e734-2f23-44db-97c1-504000002b01"
      },
      "layers": {
        "a864e734-2f23-44db-97c1-504000002b00": {
          "class": "com.simiacryptus.mindseye.layers.java.ImgCropLayer",
          "id": "a864e734-2f23-44db-97c1-504000002b00",
          "isFrozen": false,
          "name": "ImgCropLayer/a864e734-2f23-44db-97c1-504000002b00",
          "sizeX": 4,
          "sizeY": 4
        },
        "a864e734-2f23-44db-97c1-504000002b01": {
          "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
          "id": "a864e734-2f23-44db-97c1-504000002b01",
          "isFrozen": false,
          "name": "MeanSqLossLayer/a864e734-2f23-44db-97c1-504000002b01"
        }
      },
      "links": {
        "848b0b20-a4eb-4aaf-b695-d5a17b2abc82": [
          "f58be311-6abc-4576-b9d6-640ea38e9d53"
        ],
        "23bbba8a-a406-467d-8ee6-925057fc17a0": [
          "848b0b20-a4eb-4aaf-b695-d5a17b2abc82",
          "f58be311-6abc-4576-b9d6-640ea38e9d53"
        ]
      },
      "labels": {},
      "head": "23bbba8a-a406-467d-8ee6-925057fc17a0"
    }
```



### Network Diagram
Code from [LayerTestBase.java:94](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L94) executed in 0.14 seconds: 
```java
    return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.1.png)



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.192, -0.012, -0.808 ], [ 0.188, -1.52, -0.312 ], [ -0.636, 0.96, -0.096 ], [ 0.956, -0.212, 1.88 ] ],
    	[ [ -0.556, -0.88, 0.64 ], [ 1.492, 0.18, 0.052 ], [ 0.344, 1.448, 0.488 ], [ 1.032, 1.188, 0.98 ] ],
    	[ [ -0.324, 0.12, 1.716 ], [ 0.796, -0.32, 0.228 ], [ -0.02, 1.884, -1.392 ], [ 1.116, 0.068, -0.412 ] ],
    	[ [ 1.664, -1.52, -1.012 ], [ 1.98, 1.012, 1.548 ], [ 1.84, -0.888, -0.28 ], [ 0.256, 0.132, -1.572 ] ]
    ],
    [
    	[ [ 0.652, 0.476, -0.948 ], [ -1.024, -1.196, -0.88 ], [ -0.076, -1.236, -0.84 ], [ -0.412, -1.448, -0.948 ], [ 0.016, 0.368, -0.352 ] ],
    	[ [ 1.356, 0.8, 0.968 ], [ -1.748, -1.096, -1.564 ], [ 1.856, 1.556, 0.924 ], [ 0.024, -1.884, -1.04 ], [ -1.084, -0.284, -0.912 ] ],
    	[ [ 1.44, -0.444, -1.516 ], [ -0.532, -1.512, 0.128 ], [ 1.276, -1.912, 1.912 ], [ -0.348, -0.252, 1.12 ], [ 0.036, -0.612, 0.252 ] ],
    	[ [ -1.012, -1.816, -1.496 ], [ 1.9, 0.38, -0.728 ], [ 0.508, 0.412, -1.212 ], [ 1.74, 0.008, -1.284 ], [ -1.74, 1.708, 1.488 ] ],
    	[ [ 0.352, -0.568, 1.188 ], [ 0.544, -0.708, -1.316 ], [ 1.288, -1.716, -0.308 ], [ 1.016, -1.62, -0.044 ], [ 0.144, 0.372, -0.492 ] ]
    ]]
    --------------------
    Output: 
    [ 0.0 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1240#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.192, -0.012, -0.808 ], [ 0.188, -1.52, -0.312 ], [ -0.636, 0.96, -0.096 ], [ 0.956, -0.212, 1.88 ] ],
    	[ [ -0.556, -0.88, 0.64 ], [ 1.492, 0.18, 0.052 ], [ 0.344, 1.448, 0.488 ], [ 1.032, 1.188, 0.98 ] ],
    	[ [ -0.324, 0.12, 1.716 ], [ 0.796, -0.32, 0.228 ], [ -0.02, 1.884, -1.392 ], [ 1.116, 0.068, -0.412 ] ],
    	[ [ 1.664, -1.52, -1.012 ], [ 1.98, 1.012, 1.548 ], [ 1.84, -0.888, -0.28 ], [ 0.256, 0.132, -1.572 ] ]
    ],
    [
    	[ [ 0.652, 0.476, -0.948 ], [ -1.024, -1.196, -0.88 ], [ -0.076, -1.236, -0.84 ], [ -0.412, -1.448, -0.948 ], [ 0.016, 0.368, -0.352 ] ],
    	[ [ 1.356, 0.8, 0.968 ], [ -1.748, -1.096, -1.564 ], [ 1.856, 1.556, 0.924 ], [ 0.024, -1.884, -1.04 ], [ -1.084, -0.284, -0.912 ] ],
    	[ [ 1.44, -0.444, -1.516 ], [ -0.532, -1.512, 0.128 ], [ 1.276, -1.912, 1.912 ], [ -0.348, -0.252, 1.12 ], [ 0.036, -0.612, 0.252 ] ],
    	[ [ -1.012, -1.816, -1.496 ], [ 1.9, 0.38, -0.728 ], [ 0.508, 0.412, -1.212 ], [ 1.74, 0.008, -1.284 ], [ -1.74, 1.708, 1.488 ] ],
    	[ [ 0.352, -0.568, 1.188 ], [ 0.544, -0.708, -1.316 ], [ 1.288, -1.716, -0.308 ], [ 1.016, -1.62, -0.044 ], [ 0.144, 0.372, -0.492 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2965882619576667, negative=19, min=-1.572, max=-1.572, mean=0.28350000000000003, count=48.0, positive=29, stdDev=0.9763696789638646, zeros=0},
    {meanExponent=-0.19236098259330195, negative=42, min=-0.492, max=-0.492, mean=-0.18602666666666665, count=75.0, positive=33, stdDev=1.0839659277958058, zeros=0}
    Output: [ 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.192, -0.012, -0.808 ], [ 0.188, -1.52, -0.312 ], [ -0.636, 0.96, -0.096 ], [ 0.956, -0.212, 1.88 ] ],
    	[ [ -0.556, -0.88, 0.64 ], [ 1.492, 0.18, 0.052 ], [ 0.344, 1.448, 0.488 ], [ 1.032, 1.188, 0.98 ] ],
    	[ [ -0.324, 0.12, 1.716 ], [ 0.796, -0.32, 0.228 ], [ -0.02, 1.884, -1.392 ], [ 1.116, 0.068, -0.412 ] ],
    	[ [ 1.664, -1.52, -1.012 ], [ 1.98, 1.012, 1.548 ], [ 1.84, -0.888, -0.28 ], [ 0.256, 0
```
...[skipping 591 bytes](etc/1.txt)...
```
     max=0.0, mean=0.0, count=48.0, positive=0, stdDev=0.0, zeros=48}
    Feedback Error: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=48.0, positive=0, stdDev=0.0, zeros=48}
    Feedback for input 1
    Inputs Values: [
    	[ [ 0.652, 0.476, -0.948 ], [ -1.024, -1.196, -0.88 ], [ -0.076, -1.236, -0.84 ], [ -0.412, -1.448, -0.948 ], [ 0.016, 0.368, -0.352 ] ],
    	[ [ 1.356, 0.8, 0.968 ], [ -1.748, -1.096, -1.564 ], [ 1.856, 1.556, 0.924 ], [ 0.024, -1.884, -1.04 ], [ -1.084, -0.284, -0.912 ] ],
    	[ [ 1.44, -0.444, -1.516 ], [ -0.532, -1.512, 0.128 ], [ 1.276, -1.912, 1.912 ], [ -0.348, -0.252, 1.12 ], [ 0.036, -0.612, 0.252 ] ],
    	[ [ -1.012, -1.816, -1.496 ], [ 1.9, 0.38, -0.728 ], [ 0.508, 0.412, -1.212 ], [ 1.74, 0.008, -1.284 ], [ -1.74, 1.708, 1.488 ] ],
    	[ [ 0.352, -0.568, 1.188 ], [ 0.544, -0.708, -1.316 ], [ 1.288, -1.716, -0.308 ], [ 1.016, -1.62, -0.044 ], [ 0.144, 0.372, -0.492 ] ]
    ]
    Value Statistics: {meanExponent=-0.19236098259330195, negative=42, min=-0.492, max=-0.492, mean=-0.18602666666666665, count=75.0, positive=33, stdDev=1.0839659277958058, zeros=0}
    Implemented Feedback: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], ... ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=75.0, positive=0, stdDev=0.0, zeros=75}
    Measured Feedback: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], ... ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=75.0, positive=0, stdDev=0.0, zeros=75}
    Feedback Error: [ [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], [ 0.0 ], ... ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=75.0, positive=0, stdDev=0.0, zeros=75}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (123#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (123#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.4608 +- 0.1003 [0.3648 - 1.2511]
    Learning performance: 0.0980 +- 0.0350 [0.0655 - 0.2422]
    
```

