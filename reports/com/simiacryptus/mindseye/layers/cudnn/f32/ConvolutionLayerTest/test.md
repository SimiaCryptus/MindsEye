# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa4500000036",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa4500000036",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          0.296,
          1.544,
          0.508,
          -1.256
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.01 seconds: 
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
    	[ [ 0.128, -1.968 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -0.9618560671806335, 2.669440269470215 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:131](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L131) executed in 0.02 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "370a9587-74a1-4959-b406-fa450000003d",
      "isFrozen": false,
      "name": "ConvolutionLayer/370a9587-74a1-4959-b406-fa450000003d",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          0.296,
          1.544,
          0.508,
          -1.256
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ 0.128, -1.968 ] ]
    ]
    Error: [
    	[ [ -6.718063350064085E-8, 2.6947021503076485E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 1.6833e-07 +- 1.0114e-07 [6.7181e-08 - 2.6947e-07] (2#)
    relativeTol: 4.2698e-08 +- 7.7754e-09 [3.4922e-08 - 5.0473e-08] (2#)
    
```

### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.03 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.03 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.128, -1.968 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.2993824681284045, negative=1, min=-1.968, max=-1.968, mean=-0.9199999999999999, count=2.0, positive=1, stdDev=1.048, zeros=0}
    Output: [
    	[ [ -0.9618560671806335, 2.669440269470215 ] ]
    ]
    Outputs Statistics: {meanExponent=0.20476514821887876, negative=1, min=2.669440269470215, max=2.669440269470215, mean=0.8537921011447906, count=2.0, positive=1, stdDev=1.8156481683254242, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.128, -1.968 ] ]
    ]
    Value Statistics: {meanExponent=-0.2993824681284045, negative=1, min=-1.968, max=-1.968, mean=-0.9199999999999999, count=2.0, positive=1, stdDev=1.048, zeros=0}
    Implemented Feedback: [ [ 0.29600000381469727, 1.5440000295639038 ], [ 0.5080000162124634, -1.25600004196167 ] ]
    Implemented Statistics: {meanExponent=-0.13380189974351156, negative=1, min=-1.25600004196167, max=-1.25600004196167, mean=0.27300000190734863, count=4.0, positive=3, stdDev=1.0011288892127685, zeros=0}
    Measured Feedback: [ [ 0.2962350845336914, 1.5425682067871094 ], [ 0.508427619934082, -1.2564659118652344 ] ]
    Measured Statistics: {meanExponent=-0.13368482132604198, negative=1, min=-1.2564659118652344, max=-1.2564659118652344, mean=0.2726912498474121, count=4.0, positive=3, stdDev=1.00087899609754, zeros=0}
    Feedback Error: [ [ 2.3508071899414062E-4, -0.0014318227767944336 ], [ 4.2760372161865234E-4, -4.658699035644531E-4 ] ]
    Error Statistics: {meanExponent=-3.293396897947061, negative=2, min=-4.658699035644531E-4, max=-4.658699035644531E-4, mean=-3.0875205993652344E-4, count=4.0, positive=2, stdDev=7.286885103250931E-4, zeros=0}
    Learning Gradient for weight set 0
    Weights: [ 0.296, 1.544, 0.508, -1.256 ]
    Implemented Gradient: [ [ 0.12800000607967377, 0.0 ], [ 0.0, 0.12800000607967377 ], [ -1.968000054359436, 0.0 ], [ 0.0, -1.968000054359436 ] ]
    Implemented Statistics: {meanExponent=-0.29938245181649603, negative=2, min=-1.968000054359436, max=-1.968000054359436, mean=-0.46000001206994057, count=8.0, positive=2, stdDev=0.8722110081708665, zeros=4}
    Measured Gradient: [ [ 0.12814998626708984, 0.0 ], [ 0.0, 0.12636184692382812 ], [ -1.9669532775878906, 0.0 ], [ 0.0, -1.9693374633789062 ] ]
    Measured Statistics: {meanExponent=-0.30063782109290144, negative=2, min=-1.9693374633789062, max=-1.9693374633789062, mean=-0.46022236347198486, count=8.0, positive=2, stdDev=0.8721487821396752, zeros=4}
    Gradient Error: [ [ 1.4998018741607666E-4, 0.0 ], [ 0.0, -0.001638159155845642 ], [ 0.0010467767715454102, 0.0 ], [ 0.0, -0.0013374090194702148 ] ]
    Error Statistics: {meanExponent=-3.1158729224803006, negative=2, min=-0.0013374090194702148, max=-0.0013374090194702148, mean=-2.2235140204429626E-4, count=8.0, positive=2, stdDev=8.058336369388743E-4, zeros=4}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 5.6106e-04 +- 5.9988e-04 [0.0000e+00 - 1.6382e-03] (12#)
    relativeTol: 1.1373e-03 +- 2.0076e-03 [1.8542e-04 - 6.4403e-03] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=5.6106e-04 +- 5.9988e-04 [0.0000e+00 - 1.6382e-03] (12#), relativeTol=1.1373e-03 +- 2.0076e-03 [1.8542e-04 - 6.4403e-03] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.28 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 10.0341 +- 1.0203 [8.8030 - 15.4316]
    Learning performance: 6.9618 +- 1.2680 [5.8791 - 15.9759]
    
```

