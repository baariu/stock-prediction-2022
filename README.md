# stock-prediction-2022
 1. A FEW CHALLENGES I ENCOUNTERED. SOME OUT OF SHEER CARELESSNESS BUT YOU DON'T HAVE TO. NEWER HEROKU PROCESSES USE git push heroku main and not git push heroku master.
 2. ANOTHER PITFALL THAT ALMOST HAD ME,I WOULD GIT ADD ALL MY FILES EVEN THE VENV AND GIT WOULD GO HAYWIRE. SO, I CREATED A .gitignore FILE AND ADDED .env INSIDE THE     FOLDER SO THAT EVEN WHEN I STAGE ALL MY FILES ON GIT,IT WILL ALWAYS IGNORE MY VIRTUAL ENVIRONMENT(VENV) FILE.
 3. ENSURE YOUR PROCFILE IS WELL SPELLED. IF NOT HEROKU WILL KEEP THROWING YOU ERRORS EVERYTIME YOU TRY TO RUN A DYNO. SPELLING IS Procfile. ONLY THE 'P' IS CAPITAL
 4. THIS IS THE LINK TO THE APP ON HEROKU https://tranquil-chamber-53112.herokuapp.com/


The project involves using data from yahoo finance to train a model that can then be used to make predictions of future stock prices. A web application is also built and deployed on heroku giving users an opportunity to interact with the ML model and make predictions and also check historical data of different stocks.
