QT += core
QT -= gui

TARGET = RayTracing
CONFIG += console
CONFIG -= app_bundle
CONFIG += c++11

QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp

TEMPLATE = app

SOURCES += main.cpp

