# L1NormalizationLayer
## L1NormalizationLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.L1NormalizationLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d000014f2",
      "isFrozen": false,
      "name": "L1NormalizationLayer/e2d0bffa-47dc-4875-864f-3d3d000014f2"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[ 0.6884955752212389, 0.3203539823008849, -0.6407079646017698, 0.6318584070796459 ]]
    --------------------
    Output: 
    [ 0.6884955752212389, 0.3203539823008849, -0.6407079646017698, 0.6318584070796459 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ 0.6884955752212389, 0.3203539823008849, -0.6407079646017698, 0.6318584070796459 ]
    Output: [ 0.6884955752212389, 0.3203539823008849, -0.6407079646017698, 0.6318584070796459 ]
    Measured: [ [ 0.3114732774511797, -0.3203219501057841, 0.6406439002115683, -0.6317952275569638 ], [ -0.6884267325479865, 0.6795780598928269, 0.6406439002115683, -0.6317952275569638 ], [ -0.6884267325479865, -0.3203219501057841, 1.6405439102107344, -0.6317952275569638 ], [ -0.6884267325479865, -0.3203219501057841, 0.6406439002115683, 0.36810478244220235 ] ]
    Implemented: [ [ 0.311504424778761, -0.320353982300885, 0.64070796460177, -0.6318584070796461 ], [ -0.6884955752212392, 0.6796460176991151, 0.64070796460177, -0.6318584070796461 ], [ -0.6884955752212392, -0.320353982300885, 1.6407079646017702, -0.6318584070796461 ], [ -0.6884955752212392, -0.320353982300885, 0.64070796460177, 0.36814159292035403 ] ]
    Error: [ [ -3.114732758130456E-5, 3.2032195100895056E-5, -6.406439020179011E-5, 6.317952268231064E-5 ], [ 6.884267325268922E-5, -6.795780628821024E-5, -6.406439020179011E-5, 6.317952268231064E-5 ], [ 6.884267325268922E-5, 3.2032195100895056E-5, -1.6405439103572839E-4, 6.317952268231064E-5 ], [ 6.884267325268922E-5, 3.2032195100895056E-5, -6.406439020179011E-5, -3.6810478151683146E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 6.1520e-05 +- 3.0501e-05 [3.1147e-05 - 1.6405e-04] (16#)
    relativeTol: 4.9998e-05 +- 6.4311e-13 [4.9997e-05 - 4.9998e-05] (16#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1321 +- 0.0273 [0.0969 - 0.2394]
    Learning performance: 0.0031 +- 0.0019 [0.0000 - 0.0114]
    
```

