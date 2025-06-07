const PRESETS = [
  { id: 1, title: "Машина", preview: "previews/car.jpg", model: "models/car.glb" },
  { id: 2, title: "Дерево", preview: "previews/tree.jpg", model: "models/tree.glb" },
  { id: 3, title: "Дом", preview: "previews/house.jpg", model: "models/house.glb" },
  { id: 4, title: "Мост", preview: "previews/bridge.jpg", model: "models/bridge.glb" },
  { id: 5, title: "Машина + Дерево", models: [
      { preview: "previews/car.jpg", model: "models/car.glb" },
      { preview: "previews/tree.jpg", model: "models/tree.glb" },
    ]
  }
];

let userModels = []; // user models from IndexedDB
const cardList = document.getElementById('card-list');

// Загружаем пользовательские модели из IndexedDB при загрузке страницы
getAllModelsFromDB().then(models => {
  userModels = models;
  renderCards();
});

// Генерация карточек
function renderCards() {
  cardList.innerHTML = '';
  PRESETS.forEach(model => addCard(model, false));
  userModels.forEach(model => addCard(model, true, model.id));
}

function addCard(model, isUser, userId) {
  const card = document.createElement('div');
  card.className = 'card';
  card.tabIndex = 0;

  if (model.models) {
    // Карточка с двумя моделями
    const pair = document.createElement('div');
    pair.className = 'model-pair';
    model.models.forEach(submodel => {
      // Для простоты ставим заглушку
      const img = document.createElement('div');
      img.className = 'preview-canvas';
      img.style.display = 'flex';
      img.style.justifyContent = 'center';
      img.style.alignItems = 'center';
      img.style.fontSize = '2rem';
      img.textContent = '🧩';
      pair.appendChild(img);
    });
    card.appendChild(pair);
  } else if (model.preview) {
    // Предустановленные — картинка-превью (можно заменить на заглушку)
    const img = document.createElement('img');
    img.src = model.preview;
    img.onerror = function() {
      img.style.display = "none";
      const fallback = document.createElement('div');
      fallback.className = 'preview-canvas';
      fallback.style.display = 'flex';
      fallback.style.justifyContent = 'center';
      fallback.style.alignItems = 'center';
      fallback.style.fontSize = '2rem';
      fallback.textContent = '🧩';
      card.insertBefore(fallback, title);
    };
    img.className = 'preview-canvas';
    card.appendChild(img);
  } else if (isUser) {
    // Пользовательская модель — просто заглушка
    const div = document.createElement('div');
    div.className = 'preview-canvas';
    div.style.display = 'flex';
    div.style.justifyContent = 'center';
    div.style.alignItems = 'center';
    div.style.fontSize = '2rem';
    div.style.color = '#888';
    div.textContent = '🧩';
    card.appendChild(div);
  }

  const title = document.createElement('div');
  title.className = 'card-title';
  title.textContent = model.title || 'Загруженная модель';
  card.appendChild(title);

  card.onclick = () => {
    if (isUser) {
      window.location.href = `detail.html?user=${userId}`;
    } else {
      window.location.href = `detail.html?id=${model.id}`;
    }
  };

  cardList.appendChild(card);
}

// Загрузка пользовательских моделей
const uploadInput = document.getElementById('uploadModel');
uploadInput.addEventListener('change', (event) => {
  const files = Array.from(event.target.files);
  files.forEach(file => {
    const reader = new FileReader();
    reader.onload = function(e) {
      const modelObj = {
        title: file.name,
        buffer: e.target.result,
        filename: file.name
      };
      addModelToDB(modelObj).then(id => {
        modelObj.id = id;
        userModels.push(modelObj);
        renderCards();
      });
    };
    reader.readAsArrayBuffer(file);
  });
});
