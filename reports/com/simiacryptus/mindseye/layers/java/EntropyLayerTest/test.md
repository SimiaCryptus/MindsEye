# EntropyLayer
## EntropyLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLayer",
      "id": "370a9587-74a1-4959-b406-fa4500002ba6",
      "isFrozen": true,
      "name": "EntropyLayer/370a9587-74a1-4959-b406-fa4500002ba6"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
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
    	[ [ 1.104 ], [ 1.488 ], [ 0.86 ] ],
    	[ [ -0.148 ], [ -0.204 ], [ -1.348 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.10922970243181364 ], [ -0.5913802093794194 ], [ 0.12970768517174194 ] ],
    	[ [ -0.28276036477226724 ], [ -0.3242855981681358 ], [ 0.4025424728366755 ] ]
    ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 1.104 ], [ 1.488 ], [ 0.86 ] ],
    	[ [ -0.148 ], [ -0.204 ], [ -1.348 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2067246281888725, negative=3, min=-1.348, max=-1.348, mean=0.2919999999999999, count=6.0, positive=3, stdDev=0.9610411021387172, zeros=0}
    Output: [
    	[ [ -0.10922970243181364 ], [ -0.5913802093794194 ], [ 0.12970768517174194 ] ],
    	[ [ -0.28276036477226724 ], [ -0.3242855981681358 ], [ 0.4025424728366755 ] ]
    ]
    Outputs Statistics: {meanExponent=-0.5849448081531003, negative=4, min=0.4025424728366755, max=0.4025424728366755, mean=-0.12923428612386975, count=6.0, positive=2, stdDev=0.322860076703722, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 1.104 ], [ 1.488 ], [ 0.86 ] ],
    	[ [ -0.148 ], [ -0.204 ], [ -1.348 ] ]
    ]
    Value Statistics: {meanExponent=-0.2067246281888725, negative=3, min=-1.348, max=-1.348, mean=0.2919999999999999, count=6.0, positive=3, stdDev=0.9610411021387172, zeros=0}
    Implemented Feedback: [ [ -1.0989399478549036, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9105430052180221, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.3974329364109002, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.5896352851379207, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.8491771102654163, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.2986220124901153 ] ]
    Implemented Statistics: {meanExponent=-0.006888349588711142, negative=4, min=-1.2986220124901153, max=-1.2986220124901153, mean=-0.08733315879626091, count=36.0, positive=2, stdDev=0.4239428731169346, zeros=30}
    Measured Feedback: [ [ -1.0989852363425812, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.9108809191710199, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -1.3974665378080342, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.5898804232357113, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.8492352475469866, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -1.2985849195840116 ] ]
    Measured Statistics: {meanExponent=-0.006823794993771465, negative=4, min=-1.2985849195840116, max=-1.2985849195840116, mean=-0.08731973885763561, count=36.0, positive=2, stdDev=0.4239816889087166, zeros=30}
    Feedback Error: [ [ -4.528848767759719E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.379139529977593E-4, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -3.360139713404209E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 2.4513809779058704E-4, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -5.813728157033626E-5, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 3.709290610376392E-5 ] ]
    Error Statistics: {meanExponent=-4.094282054723855, negative=3, min=3.709290610376392E-5, max=3.709290610376392E-5, mean=1.341993862528152E-5, count=36.0, positive=3, stdDev=6.986714450350003E-5, zeros=30}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.1033e-05 +- 6.7964e-05 [0.0000e+00 - 3.3791e-04] (36#)
    relativeTol: 7.9082e-05 +- 8.3698e-05 [1.2022e-05 - 2.0783e-04] (6#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.1033e-05 +- 6.7964e-05 [0.0000e+00 - 3.3791e-04] (36#), relativeTol=7.9082e-05 +- 8.3698e-05 [1.2022e-05 - 2.0783e-04] (6#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1303 +- 0.0283 [0.1083 - 0.2907]
    Learning performance: 0.0030 +- 0.0050 [0.0000 - 0.0456]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



