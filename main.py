from src.local_model import LocalModel

model = LocalModel(model_name_or_path="microsoft/phi-2")

prompt = \
'''
def print_prime(n):
   """
   Print all primes between 1 and n
   """
'''

print(model.predict(prompt=prompt, stop="```"))