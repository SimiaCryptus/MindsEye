# HyperbolicActivationLayer
## HyperbolicActivationLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.HyperbolicActivationLayer",
      "id": "a864e734-2f23-44db-97c1-504000002bbc",
      "isFrozen": false,
      "name": "HyperbolicActivationLayer/a864e734-2f23-44db-97c1-504000002bbc",
      "weights": {
        "dimensions": [
          2
        ],
        "data": [
          1.0,
          1.0
        ]
      },
      "negativeMode": 1
    }
```



### Reference Input/Output Pairs
Code from [LayerTestBase.java:110](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L110) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, input);
    DoubleStatistics error = new DoubleStatistics().accept(eval.getOutput().add(output.scale(-1)).getData());
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\nError: %s",
      Arrays.stream(input).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint(), error);
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.0 ]]
    --------------------
    Output: 
    [ 0.0 ]
    Error: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (120#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [
    	[ [ 0.004 ], [ -1.136 ], [ -1.82 ] ],
    	[ [ -0.548 ], [ 0.516 ], [ -0.588 ] ]
    ]
    Inputs Statistics: {meanExponent=-0.476947117187374, negative=4, min=-0.588, max=-0.588, mean=-0.5953333333333334, count=6.0, positive=2, stdDev=0.7515190542420657, zeros=0}
    Output: [
    	[ [ 7.999968000271807E-6 ], [ 0.5134384691820146 ], [ 1.076631888419322 ] ],
    	[ [ 0.14030873012531142 ], [ 0.125280409498006 ], [ 0.16006206730502126 ] ]
    ]
    Outputs Statistics: {meanExponent=-1.3175166377129346, negative=0, min=0.16006206730502126, max=0.16006206730502126, mean=0.33595492741627925, count=6.0, positive=6, stdDev=0.36664301245057235, zeros=0}
    Feedback for input 0
    Inputs Values: [
    	[ [ 0.004 ], [ -1.136 ], [ -1.82 ] ],
    	[ [ -0.548 ], [ 0.516 ], [ -0.588 ] ]
    ]
    Value Statistics: {meanExponent=-0.476947117187374, negative=4, min=-0.588, max=-0.588, mean=-0.5953333333333334, count=6.0, positive=2, stdDev=0.7515190542420657, zeros=0}
    Implemented Feedback: [ [ 0.0039999680003839945, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.48057160795373277, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.7506086458962463, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.4585523711642599, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.8764191719050104, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.5068694310176026 ] ]
    Implemented Statistics: {meanExponent=-0.5886291497416444, negative=4, min=-0.5068694310176026, max=-0.5068694310176026, mean=-0.059775458822443, count=36.0, positive=2, stdDev=0.2297972581250799, zeros=30}
    Measured Feedback: [ [ 0.004049966779717806, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, -0.48053788536117636, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, -0.7505942215146888, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.4585874600859796, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, -0.8764135883598811, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, -0.5068374018746269 ] ]
    Measured Statistics: {meanExponent=-0.5877359593626673, negative=4, min=-0.5068374018746269, max=-0.5068374018746269, mean=-0.059770713062352106, count=36.0, positive=2, stdDev=0.22979464033610358, zeros=30}
    Feedback Error: [ [ 4.999877933381178E-5, 0.0, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 3.372259255640664E-5, 0.0, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.4424381557520682E-5, 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 3.5088921719717E-5, 0.0, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 5.583545129250744E-6, 0.0 ], [ 0.0, 0.0, 0.0, 0.0, 0.0, 3.202914297573134E-5 ] ]
    Error Statistics: {meanExponent=-4.636066178284293, negative=0, min=3.202914297573134E-5, max=3.202914297573134E-5, mean=4.74576009090106E-6, count=36.0, positive=6, stdDev=1.215939025801572E-5, zeros=30}
    Learning Gradient for weight set 0
    Weights: [ 1.0, 1.0 ]
    Implemented Gradient: [ [ -0.9999920000959988, 0.0, 0.0, -0.8886673859772478, 0.0, 0.0 ], [ 0.0, -0.8769554889666656, -0.6607470474438788, 0.0, -0.4815489955522035, -0.8620228418666712 ] ]
    Implemented Statistics: {meanExponent=-0.1116820325542704, negative=6, min=-0.8620228418666712, max=-0.8620228418666712, mean=-0.39749447999188886, count=12.0, positive=0, stdDev=0.41576253128229745, zeros=6}
    Measured Gradient: [ [ -0.9998920100949821, 0.0, 0.0, -0.8885691862384193, 0.0, 0.0 ], [ 0.0, -0.8768576768322656, -0.6606623685712076, 0.0, -0.48148235459288813, -0.8619255761921352 ] ]
    Measured Statistics: {meanExponent=-0.11173280518236833, negative=6, min=-0.8619255761921352, max=-0.8619255761921352, mean=-0.39744909771015813, count=12.0, positive=0, stdDev=0.4157167419731269, zeros=6}
    Gradient Error: [ [ 9.999000101668098E-5, 0.0, 0.0, 9.81997388285194E-5, 0.0, 0.0 ], [ 0.0, 9.781213440007708E-5, 8.467887267116225E-5, 0.0, 6.664095931535607E-5, 9.726567453605117E-5 ] ]
    Error Statistics: {meanExponent=-4.046344074616896, negative=0, min=9.726567453605117E-5, max=9.726567453605117E-5, mean=4.5382281730653916E-5, count=12.0, positive=6, stdDev=4.615727198261178E-5, zeros=6}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4905e-05 +- 3.0873e-05 [0.0000e+00 - 9.9990e-05] (48#)
    relativeTol: 5.5663e-04 +- 1.7050e-03 [3.1854e-06 - 6.2111e-03] (12#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.4905e-05 +- 3.0873e-05 [0.0000e+00 - 9.9990e-05] (48#), relativeTol=5.5663e-04 +- 1.7050e-03 [3.1854e-06 - 6.2111e-03] (12#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1782 +- 0.0672 [0.1225 - 0.5500]
    Learning performance: 0.0456 +- 0.0291 [0.0370 - 0.2679]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:74](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L74) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:78](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L78) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



