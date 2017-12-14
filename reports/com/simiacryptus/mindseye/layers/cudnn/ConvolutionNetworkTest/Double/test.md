# PipelineNetwork
## Double
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.network.PipelineNetwork",
      "id": "11bcbbd0-19db-4adf-91e8-afb4a840f8c8",
      "isFrozen": false,
      "name": "PipelineNetwork/11bcbbd0-19db-4adf-91e8-afb4a840f8c8",
      "inputs": [
        "4f5aeeb5-e92c-415b-a038-fc41d74d7518"
      ],
      "nodes": {
        "2f33a5ab-1eb2-4f24-b9c7-ec12feebbd57": "1296309d-e644-4d30-bab9-d6ff42ede5d4",
        "0a6c8261-af5c-46e8-b6a1-1800f364a055": "f432cec0-19e8-421b-8943-0de0b6512aa9",
        "ac20eca4-e2ad-4ca7-acec-e6214dd636b5": "3bf87ad0-cd40-4802-965a-bef21d4437fb"
      },
      "layers": {
        "1296309d-e644-4d30-bab9-d6ff42ede5d4": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ImgConcatLayer",
          "id": "1296309d-e644-4d30-bab9-d6ff42ede5d4",
          "isFrozen": false,
          "name": "ImgConcatLayer/1296309d-e644-4d30-bab9-d6ff42ede5d4",
          "maxBands": -1
        },
        "f432cec0-19e8-421b-8943-0de0b6512aa9": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer",
          "id": "f432cec0-19e8-421b-8943-0de0b6512aa9",
          "isFrozen": false,
          "name": "ImgBandBiasLayer/f432cec0-19e8-421b-8943-0de0b6512aa9",
          "bias": [
            0.0
          ]
        },
        "3bf87ad0-cd40-4802-965a-bef21d4437fb": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.ActivationLayer",
          "id": "3bf87ad0-cd40-4802-965a-bef21d4437fb",
          "isFrozen": false,
          "name": "ActivationLayer/3bf87ad0-cd40-4802-965a-bef21d4437fb",
          "mode": 1
        }
      },
      "links": {
        "2f33a5ab-1eb2-4f24-b9c7-ec12feebbd57": [
          "4f5aeeb5-e92c-415b-a038-fc41d74d7518"
        ],
        "0a6c8261-af5c-46e8-b6a1-1800f364a055": [
          "2f33a5ab-1eb2-4f24-b9c7-ec12feebbd57"
        ],
        "ac20eca4-e2ad-4ca7-acec-e6214dd636b5": [
          "0a6c8261-af5c-46e8-b6a1-1800f364a055"
        ]
      },
      "labels": {},
      "head": "ac20eca4-e2ad-4ca7-acec-e6214dd636b5"
    }
```



### Network Diagram
Code from [StandardLayerTests.java:79](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L79) executed in 0.12 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.641.png)



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[
    	[ [ 0.28 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 0.28 ] ]
    ]
    --------------------
    Derivative: 
    [
    	[ [ 1.0 ] ]
    ]
```



[GPU Log](etc/cuda.log)

### Batch Execution
Code from [StandardLayerTests.java:101](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L101) executed in 0.02 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (20#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (16#)}
```



### Differential Validation
Code from [StandardLayerTests.java:109](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L109) executed in 0.01 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ -0.872 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.05948351506743278, negative=1, min=-0.872, max=-0.872, mean=-0.872, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Output: [
    	[ [ 0.0 ] ]
    ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Feedback for input 0
    Inputs Values: [
    	[ [ -0.872 ] ]
    ]
    Value Statistics: {meanExponent=-0.05948351506743278, negative=1, min=-0.872, max=-0.872, mean=-0.872, count=1.0, positive=0, stdDev=0.0, zeros=0}
    Implemented Feedback: [ [ 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Measured Feedback: [ [ 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Feedback Error: [ [ 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Learning Gradient for weight set 0
    Weights: [ 0.0 ]
    Implemented Gradient: [ [ 0.0 ] ]
    Implemented Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Measured Gradient: [ [ 0.0 ] ]
    Measured Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Gradient Error: [ [ 0.0 ] ]
    Error Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (2#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (2#), relativeTol=0.0000e+00 +- 0.0000e+00 [Infinity - -Infinity] (0#)}
```



### Performance
Adding performance wrappers

Code from [TestUtil.java:269](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/TestUtil.java#L269) executed in 0.00 seconds: 
```java
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).shouldRecordSignalMetrics(false));
      }
      else {
        ((MonitoringWrapperLayer) node.getLayer()).shouldRecordSignalMetrics(false);
      }
    });
```

Code from [StandardLayerTests.java:119](../../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    
```

Returns: 

```
    java.lang.RuntimeException: java.lang.RuntimeException: java.util.concurrent.ExecutionException: com.simiacryptus.mindseye.lang.ComponentException: Error evaluating layer ActivationLayer/3bf87ad0-cd40-4802-965a-bef21d4437fb
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:156)
    	at com.simiacryptus.mindseye.test.StandardLayerTests.test(StandardLayerTests.java:119)
    	at com.simiacryptus.mindseye.layers.LayerTestBase.test(LayerTestBase.java:40)
    	at sun.reflect.GeneratedMethodAccessor16.invoke(Unknown Source)
    	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    	at java.lang.reflect.Method.invoke(Method.java:4
```
...[skipping 4376 bytes](etc/65.txt)...
```
    ad.java:748)
    Caused by: com.simiacryptus.mindseye.lang.ComponentException: Error evaluating layer ImgBandBiasLayer/f432cec0-19e8-421b-8943-0de0b6512aa9
    	at com.simiacryptus.mindseye.network.LazyResult.lambda$get$0(LazyResult.java:77)
    	at java.util.HashMap.computeIfAbsent(HashMap.java:1126)
    	at com.simiacryptus.mindseye.network.LazyResult.get(LazyResult.java:73)
    	at com.simiacryptus.mindseye.network.LazyResult.get(LazyResult.java:32)
    	at com.simiacryptus.mindseye.network.InnerNode.eval(InnerNode.java:82)
    	at com.simiacryptus.mindseye.network.LazyResult.lambda$get$0(LazyResult.java:75)
    	... 10 more
    Caused by: java.lang.AssertionError: 3 != 1
    	at com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer.eval(ImgBandBiasLayer.java:91)
    	at com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer.eval(MonitoringWrapperLayer.java:148)
    	at com.simiacryptus.mindseye.network.InnerNode.eval(InnerNode.java:83)
    	at com.simiacryptus.mindseye.network.LazyResult.lambda$get$0(LazyResult.java:75)
    	... 15 more
    
```



