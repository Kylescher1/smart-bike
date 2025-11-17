from operator import attrgetter


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


people = [Person("Alice", 30), Person("Bob", 25), Person("Charlie", 35)]
# sorted_people = sorted(people, key=attrgetter('age'))
#
# for p in sorted_people:
#     print(f"{p.name}: {p.age}")
print(people[0].age)