#    Copyright 2024 Ewen Lawrence

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Module of decorators used in code base for cleaner code
"""
from functools import wraps


def add_nickname(default_name: str):
    """Decorator to add a nickname attribute to a class

    Parameters
    ----------
    default_name : str
        default value given to nickname
    """
    def deco(cls):
        def get_attr(self):
            return getattr(self, "_nickname")

        def set_attr(self, value):
            if not isinstance(value, str):
                raise TypeError("Argument 'nickname' must be str.")
            setattr(self, "_nickname", value)
        prop = property(get_attr, set_attr)
        setattr(cls, "nickname", prop)
        # Default value for nickname
        setattr(cls, "_nickname", default_name)
        return cls
    return deco


def immutable(attribute: str):
    """Decorator to make an attribute immutable

    Parameters
    ----------
    attribute : str
        name of the attribute
    """
    def _make_immutable(f):
        @wraps(f)
        def wrapper(self, *args):
            try:
                _ = getattr(self, attribute)
                raise ValueError("Argument '"+attribute+"' is immutable and "
                                 "already has a value set")
            except ValueError as e:
                raise e
            except AttributeError:
                return f(self, *args)
        return wrapper
    return _make_immutable
