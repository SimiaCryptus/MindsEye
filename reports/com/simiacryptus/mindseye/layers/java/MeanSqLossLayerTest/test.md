# MeanSqLossLayer
## MeanSqLossLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.java.MeanSqLossLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f83",
      "isFrozen": false,
      "name": "MeanSqLossLayer/c88cbdf1-1c2a-4a5e-b964-890900000f83"
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
    [[ 0.192, 1.692, -1.524, 1.12 ],
    [ 1.552, 1.696, 0.108, -1.8 ]]
    --------------------
    Output: 
    [ 3.2598599999999998 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: MeanSqLossLayer/c88cbdf1-1c2a-4a5e-b964-890900000f83
    Inputs: [ 0.192, 1.692, -1.524, 1.12 ],
    [ 1.552, 1.696, 0.108, -1.8 ]
    output=[ 3.2598599999999998 ]
    measured/actual: [ [ -0.6799749999997218 ], [ -0.001974999999099225 ], [ -0.8159749999991917 ], [ 1.460025000010745 ] ]
    implemented/expected: [ [ -0.68 ], [ -0.0020000000000000018 ], [ -0.8160000000000001 ], [ 1.46 ] ]
    error: [ [ 2.5000000278274648E-5 ], [ 2.5000000900776698E-5 ], [ 2.5000000808406142E-5 ], [ 2.5000010745124257E-5 ] ]
    Component: MeanSqLossLayer/c88cbdf1-1c2a-4a5e-b964-890900000f83
    Inputs: [ 0.192, 1.692, -1.524, 1.12 ],
    [ 1.552, 1.696, 0.108, -1.8 ]
    output=[ 3.2598599999999998 ]
    measured/actual: [ [ 0.6800249999994179 ], [ 0.0020249999987953515 ], [ 0.8160249999988878 ], [ -1.4599749999888445 ] ]
    implemented/expected: [ [ 0.68 ], [ 0.0020000000000000018 ], [ 0.8160000000000001 ], [ -1.46 ] ]
    error: [ [ 2.4999999417851804E-5 ], [ 2.4999998795349754E-5 ], [ 2.499999888772031E-5 ], [ 2.5000011155462687E-5 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.5000e-05 +- 4.8554e-12 [2.5000e-05 - 2.5000e-05] (8#)
    relativeTol: 1.5731e-03 +- 2.7004e-03 [8.5616e-06 - 6.2893e-03] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.1722 +- 0.0911 [0.1168 - 0.9233]
    Learning performance: 0.0047 +- 0.0076 [0.0000 - 0.0741]
    
```

