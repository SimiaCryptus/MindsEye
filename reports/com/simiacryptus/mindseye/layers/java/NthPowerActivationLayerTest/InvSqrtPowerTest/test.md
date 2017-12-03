# NthPowerActivationLayer
## InvSqrtPowerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer",
      "id": "ff6064d4-4ed4-46f2-9d30-74010000007a",
      "isFrozen": false,
      "name": "NthPowerActivationLayer/ff6064d4-4ed4-46f2-9d30-74010000007a",
      "power": -0.5
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ 1.616, 1.368, 0.676 ]]
    --------------------
    Output: 
    [ 0.7866459694094408, 0.8549819600709617, 1.2162606385262997 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 1.616, 1.368, 0.676 ]
    Output: [ 0.7866459694094408, 0.8549819600709617, 1.2162606385262997 ]
    Measured: [ [ -0.2433816405789102, 0.0, 0.0 ], [ 0.0, -0.31247627518826704, 0.0 ], [ 0.0, 0.0, -0.8995012684898107 ] ]
    Implemented: [ [ -0.24339293607965368, 0.0, 0.0 ], [ 0.0, -0.3124934064586848, 0.0 ], [ 0.0, 0.0, -0.8996010639987423 ] ]
    Error: [ [ 1.1295500743468967E-5, 0.0, 0.0 ], [ 0.0, 1.7131270417747313E-5, 0.0 ], [ 0.0, 0.0, 9.979550893157718E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4247e-05 +- 3.0828e-05 [0.0000e+00 - 9.9796e-05] (9#)
    relativeTol: 3.5362e-05 +- 1.4322e-05 [2.3205e-05 - 5.5470e-05] (3#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2078 +- 0.0608 [0.1596 - 0.5044]
    Learning performance: 0.0092 +- 0.0129 [0.0029 - 0.0940]
    
```

### Function Plots
Code from [ActivationLayerTestBase.java:73](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L73) executed in 0.01 seconds: 
```java
    return plot("Value Plot", plotData, x -> new double[]{x[0], x[1]});
```

Returns: 

![Result](etc/test.1.png)



Code from [ActivationLayerTestBase.java:77](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/java/ActivationLayerTestBase.java#L77) executed in 0.01 seconds: 
```java
    return plot("Derivative Plot", plotData, x -> new double[]{x[0], x[2]});
```

Returns: 

![Result](etc/test.2.png)



