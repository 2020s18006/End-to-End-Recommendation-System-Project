import sys
from books_recommender.exception.exception_handler import AppException

# Assuming your AppException class is defined above this or imported from a module

def divide_numbers(a, b):
    try:
        result = a / b
        return result
    except Exception as e:
        # Raise custom exception
        raise AppException(e, sys)

# Trigger the error
if __name__ == "__main__":
    try:
        divide_numbers(10, 0)  # Division by zero error
    except AppException as ex:
        print("Custom Exception Caught!")
        print(str(ex))  # This will call __str__ method and show detailed info
