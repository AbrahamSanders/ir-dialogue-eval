from enum import Enum

class Domain(Enum):
    #General domains
    GENERAL = "General"
    
    #SGD domains
    ALARM = "Alarm"
    BANKS = "Banks"
    BUSES = "Buses"
    CALENDAR = "Calendar"
    EVENTS = "Events"
    FLIGHTS = "Flights"
    HOMES = "Homes"
    HOTELS = "Hotels"
    MEDIA = "Media"
    MESSAGING = "Messaging"
    MOVIES = "Movies"
    MUSIC = "Music"
    PAYMENT = "Payment"
    RENTALCARS = "RentalCars"
    RESTAURANTS = "Restaurants"
    RIDESHARING = "RideSharing"
    SERVICES = "Services"
    TRAIN = "Train"
    TRAVEL = "Travel"
    WEATHER = "Weather"
    
    #MultiWOZ 2.2 domains not already in SGD
    ATTRACTION = "Attraction"
    TAXI = "Taxi"
    HOSPITAL = "Hospital"
    POLICE = "Police"