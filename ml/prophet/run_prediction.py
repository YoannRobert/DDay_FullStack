from model_prophet import ConsumptionPrediction

try:
    CP = ConsumptionPrediction()
    CP.run()
except Exception as e:
    print(e)