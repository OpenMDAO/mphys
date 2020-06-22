# from abc import ABCMeta, abstractmethod

# class Builder(metaclass=ABCMeta):
#     """
#      this is a base class intended to be used to create derived classes
#      which are builders for particular solvers

#      All of the abstract methods listed below must be implemented in any 
#      derived classes
#     """


#     def __init__(self, options):
#          self.options = options
#          self.object_built = False

#     @abstractmethod
#     def build_object(self, comm):
#         """ contracts solver/transfer scheme/etc using provided comm which
#         entails the allocation of memory for any computation"""
#         pass

#     @abstractmethod
#     def get_object(self):
#         pass

#     @abstractmethod
#     def get_component(self, **kwargs):
#         pass


class Builder(object):
    """
     this is a base class intended to be used to create derived classes
     which are builders for particular solvers

     All of the abstract methods listed below must be implemented in any 
     derived classes
    """


    def __init__(self, options):
         self.options = options
         self.object_built = False

    def build_object(self, comm):
        """ contracts solver/transfer scheme/etc using provided comm which
        entails the allocation of memory for any computation"""
        pass

    def get_object(self):
        pass

    def get_component(self, **kwargs):
        pass

